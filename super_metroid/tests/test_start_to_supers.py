"""Unit tests for continuous tip catalog + Supers report structure (no emulator)."""

from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from super_metroid.paths import ROOM_TIMINGS_DIR
from super_metroid.progression import RoomNode, RoomProgressionGraph
from super_metroid.ram import (
    ADDR_DOOR_TRANSITION,
    ADDR_GAME_STATE,
    ADDR_ROOM_ID,
    parse_state,
)
from super_metroid.room_timer import RoomTimer
from super_metroid.routes.catalog import (
    CONTINUOUS_TIPS,
    DEFAULT_CONTINUOUS_TIP,
    get_continuous_tip,
)
from super_metroid.routes.continuous import (
    CONTROLLER_PATH,
    SupersRunReport,
    default_artifact_paths,
    default_tip_artifact_paths,
    default_tip_room_timing_path,
    run_start_to_supers,
    run_to,
    write_room_timing_artifact,
)
from super_metroid.routes.runtime import RouteSession


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def test_default_artifact_paths() -> None:
    """Primary continuous tip is Varia Suit (KPDR K3)."""
    video, report = default_artifact_paths()
    assert video.name == "start_to_varia.mp4"
    assert report.name == "start_to_varia.json"
    assert DEFAULT_CONTINUOUS_TIP == "varia"


def test_continuous_tips_chain_ends_at_default() -> None:
    ids = [t.tip_id for t in CONTINUOUS_TIPS]
    assert ids == [
        "morph",
        "bombs",
        "spore",
        "supers",
        "red_tower",
        "bat",
        "below_spazer",
        "warehouse",
        "hijump",
        "kraid",
        "varia",
    ]
    assert CONTINUOUS_TIPS[-1].tip_id == DEFAULT_CONTINUOUS_TIP


def test_get_continuous_tip_aliases() -> None:
    assert get_continuous_tip("k1").tip_id == "red_tower"
    assert get_continuous_tip("start_to_supers").tip_id == "supers"
    assert get_continuous_tip("RED-TOWER").tip_id == "red_tower"
    assert get_continuous_tip("bat_room").tip_id == "bat"
    assert get_continuous_tip("k2_0").tip_id == "bat"
    assert get_continuous_tip("below").tip_id == "below_spazer"
    assert get_continuous_tip("k2_1").tip_id == "below_spazer"
    assert get_continuous_tip("warehouse_entrance").tip_id == "warehouse"
    assert get_continuous_tip("k2_6").tip_id == "warehouse"


def test_default_tip_room_timing_path() -> None:
    path = default_tip_room_timing_path("supers")
    assert path.parent == ROOM_TIMINGS_DIR
    assert path.name == "start_to_supers_room_timing.json"
    red = default_tip_room_timing_path("red_tower")
    assert red.name == "start_to_red_tower_room_timing.json"
    bat = default_tip_room_timing_path("bat")
    assert bat.name == "start_to_bat_room_timing.json"
    below = default_tip_room_timing_path("below_spazer")
    assert below.name == "start_to_below_spazer_room_timing.json"
    warehouse = default_tip_room_timing_path("warehouse")
    assert warehouse.name == "start_to_warehouse_room_timing.json"


def test_default_tip_artifact_paths_per_milestone() -> None:
    video, report = default_tip_artifact_paths("supers")
    assert video.name == "start_to_supers.mp4"
    assert report.name == "start_to_supers.json"


def test_run_to_rejects_room_timing_on_early_tip() -> None:
    with pytest.raises(ValueError, match="room timing"):
        run_to("morph", room_timing_path="/tmp/x.json")


def test_run_start_to_supers_accepts_room_timing_path() -> None:
    sig = inspect.signature(run_start_to_supers)
    assert "room_timing_path" in sig.parameters
    assert sig.parameters["room_timing_path"].default is None
    assert "tip" in inspect.signature(run_to).parameters


def test_post_supers_report_kind_keeps_evidence_fields() -> None:
    """red_tower+ kinds must serialize boss/super_collect like supers."""
    report = SupersRunReport(
        schema_version=1,
        success=True,
        outcome="warehouse_entry",
        kind="warehouse",
        error=None,
        total_frames=1,
        encoded_frames=0,
        final_state={},
        splits=[],
        progress_events=[],
        transitions=[],
        segments=[],
        boss=None,
        super_collect=None,
        action_reasons=Counter(),
        assist={},
        integrity={},
        route_plan={"id": "plan"},
        policy_sources={"mod": {}},
        state_loads=0,
        progression_writes=0,
        video=None,
        source_policy="test",
        rom_sha256="",
        start_state="power_on",
        generated_at="",
    )
    payload = report.to_dict()
    assert payload["route_plan"] == {"id": "plan"}
    assert payload["policy_sources"] == {"mod": {}}
    assert "boss" in payload
    assert "super_collect" in payload


def test_warehouse_hops_table_shape() -> None:
    from super_metroid.routes.continuous import _WAREHOUSE_HOPS

    assert [h.split_id for h in _WAREHOUSE_HOPS] == [
        "below_spazer_to_west",
        "west_to_glass",
        "glass_to_east",
        "east_to_warehouse",
    ]
    assert _WAREHOUSE_HOPS[-1].to_room == 0xA6A1


def test_controller_module_exists() -> None:
    assert CONTROLLER_PATH.is_file()


def test_supers_report_includes_super_collect_field() -> None:
    # Smoke: dataclass accepts super_collect=None for failed early exits.
    report = SupersRunReport(
        schema_version=1,
        success=False,
        outcome="failed:test",
        kind="supers",
        error="test",
        total_frames=0,
        encoded_frames=0,
        final_state={},
        splits=[],
        progress_events=[],
        transitions=[],
        segments=[],
        boss=None,
        super_collect=None,
        action_reasons=Counter(),
        assist={},
        integrity={},
        route_plan={},
        policy_sources={},
        state_loads=0,
        progression_writes=0,
        video=None,
        source_policy="test",
        rom_sha256="",
        start_state="power_on",
        generated_at="",
    )
    payload = report.to_dict()
    assert "super_collect" in payload
    assert payload["super_collect"] is None


class _NullAssist:
    telemetry = type(
        "T",
        (),
        {
            "progression_writes": 0,
            "capacity_writes": 0,
            "deaths": 0,
        },
    )()

    def apply(self, data: object, state: object) -> None:
        return None

    def report(self) -> dict[str, object]:
        return {}


class _FakeEnv:
    """Minimal env that walks a fixed RAM sequence (no ROM)."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        assert frames
        self._frames = frames
        self._index = 0
        self.data = object()

    def get_ram(self) -> np.ndarray:
        return self._frames[self._index]

    def step(self, action: object) -> tuple[np.ndarray, float, bool, bool, dict]:
        del action
        if self._index + 1 < len(self._frames):
            self._index += 1
        obs = np.zeros((2, 2, 3), dtype=np.uint8)
        return obs, 0.0, False, False, {}


def _ram_room(
    room_id: int,
    *,
    game_state: int = 8,
    door: int = 0,
) -> np.ndarray:
    ram = np.zeros(0x10000, dtype=np.uint8)
    _put_u16(ram, ADDR_GAME_STATE, game_state)
    _put_u16(ram, ADDR_ROOM_ID, room_id)
    _put_u16(ram, ADDR_DOOR_TRANSITION, door)
    return ram


def test_route_session_opt_in_room_timer_records_hop(tmp_path: Path) -> None:
    """RouteSession observes the shared RoomTimer only when attached."""
    room_a, room_b = 0x9AD9, 0x9B5B
    frames = [
        _ram_room(room_a),  # frame 0 settle
        _ram_room(room_a),  # frame 1 dwell
        _ram_room(room_a, game_state=9, door=1),  # frame 2 leave
        _ram_room(room_b),  # frame 3 settle dest
    ]
    graph = RoomProgressionGraph(
        (
            RoomNode(room_a, "A", "Brinstar"),
            RoomNode(room_b, "B", "Brinstar"),
        ),
        (),
        (),
        graph_id="synthetic_timer",
    )
    timer = RoomTimer()
    session = RouteSession(
        _FakeEnv(frames),
        writer=None,
        assist=_NullAssist(),
        graph=graph,
        room_timer=timer,
    )
    assert timer.report()["open_visit"] is not None
    idle = np.zeros(12, dtype=np.int8)
    session.step(idle, "test")
    session.step(idle, "test")
    session.step(idle, "test")
    assert len(timer.visits) == 1
    visit = timer.visits[0]
    assert visit.room_id == room_a
    assert visit.dest_room_id == room_b
    assert visit.entry_frame == 0
    assert visit.leave_frame == 2
    assert visit.exit_frame == 3

    out = tmp_path / "timing.json"
    artifact = write_room_timing_artifact(
        timer,
        path=out,
        source="test_route_session",
        route_outcome="synthetic",
        total_frames=session.frame,
        success=True,
    )
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["kind"] == "super_metroid_room_timing"
    assert loaded["visit_count"] == 1
    assert loaded["extra"]["mode"] == "continuous_route"
    assert artifact["total_room_frames"] == visit.room_frames


def test_route_session_without_timer_is_untouched() -> None:
    """Default continuous path must not invent a RoomTimer."""
    frames = [_ram_room(0x91F8), _ram_room(0x91F8)]
    graph = RoomProgressionGraph(
        (RoomNode(0x91F8, "Landing", "Crateria"),),
        (),
        (),
        graph_id="no_timer",
    )
    session = RouteSession(
        _FakeEnv(frames),
        writer=None,
        assist=_NullAssist(),
        graph=graph,
    )
    assert session.room_timer is None
    session.step(np.zeros(12, dtype=np.int8), "test")
    assert session.room_timer is None
    # Sanity: session still parses state.
    assert parse_state(frames[1], frame=1).room_id == 0x91F8
