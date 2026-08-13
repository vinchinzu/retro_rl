"""Unit tests for continuous tip catalog + Supers report structure (no emulator)."""

from __future__ import annotations

import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from super_metroid.paths import ROOM_TIMINGS_DIR
from super_metroid.progression import ObservedTransition, RoomNode, RoomProgressionGraph
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
    default_artifact_paths,
    default_tip_artifact_paths,
    default_tip_room_timing_path,
    run_supers,
    run_tip,
    run_to,
    write_room_timing_artifact,
)
from super_metroid.routes.runtime import (
    ContinuousRunReport,
    RouteSession,
    split_for_transition,
)


def _put_u16(ram: np.ndarray, address: int, value: int) -> None:
    ram[address] = value & 0xFF
    ram[address + 1] = (value >> 8) & 0xFF


def test_default_artifact_paths() -> None:
    """Primary continuous tip is Ice Beam (KPDR K4 Ice) after rr-kxge dual."""
    video, report = default_artifact_paths()
    assert video.name == "ice.mp4"
    assert report.name == "ice.json"
    assert DEFAULT_CONTINUOUS_TIP == "ice"


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
        "business",
        "frog",
        "bat_cave",
        "speed",
        "wave",
        "ice",
    ]
    # Default tip is furthest STATUS-promoted integrity-green tip (Ice).
    assert DEFAULT_CONTINUOUS_TIP == "ice"
    assert DEFAULT_CONTINUOUS_TIP in ids
    assert ids.index("speed") < ids.index("wave")
    assert ids.index("wave") < ids.index("ice")
    assert ids[-1] == DEFAULT_CONTINUOUS_TIP


def test_continuous_tips_align_with_tip_specs() -> None:
    """CLI ContinuousTip ids must match ordered TipSpec product tips (no drift)."""
    from super_metroid.routes.catalog import (
        CONTINUOUS_TIP_ORDER,
        get_named_route,
        continuous_tip_from_spec,
    )
    from super_metroid.routes.tips import TIP_BY_ID, TIP_SPECS

    catalog_ids = [t.tip_id for t in CONTINUOUS_TIPS]
    tip_spec_ids = [s.tip_id for s in TIP_SPECS]
    assert catalog_ids == list(CONTINUOUS_TIP_ORDER)
    assert catalog_ids == tip_spec_ids
    assert list(TIP_BY_ID) == catalog_ids
    # ContinuousTip / NamedRoute are projections of TipSpec CLI fields.
    for spec in TIP_SPECS:
        tip = continuous_tip_from_spec(spec)
        assert tip.display_name == spec.display_name
        assert tip.aliases == spec.aliases
        assert tip.supports_room_timing == spec.supports_room_timing
        assert tip.supports_checkpoint == spec.supports_checkpoint
        route = get_named_route(spec.tip_id)
        assert route.route_id == f"sm_{spec.tip_id}"
        assert route.display_name == spec.display_name
        assert [m.milestone_id for m in route.milestones] == list(spec.required_splits)


def test_get_continuous_tip_aliases() -> None:
    assert get_continuous_tip("k1").tip_id == "red_tower"
    assert get_continuous_tip("RED-TOWER").tip_id == "red_tower"
    assert get_continuous_tip("bat_room").tip_id == "bat"
    assert get_continuous_tip("k2_0").tip_id == "bat"
    assert get_continuous_tip("below").tip_id == "below_spazer"
    assert get_continuous_tip("k2_1").tip_id == "below_spazer"
    assert get_continuous_tip("warehouse_entrance").tip_id == "warehouse"
    assert get_continuous_tip("k2_6").tip_id == "warehouse"
    assert get_continuous_tip("k3_return").tip_id == "business"
    assert get_continuous_tip("norfair_bat").tip_id == "bat_cave"
    assert get_continuous_tip("k4_4").tip_id == "bat_cave"


def test_default_tip_room_timing_path() -> None:
    path = default_tip_room_timing_path("supers")
    assert path.parent == ROOM_TIMINGS_DIR
    assert path.name == "supers_room_timing.json"
    red = default_tip_room_timing_path("red_tower")
    assert red.name == "red_tower_room_timing.json"
    bat = default_tip_room_timing_path("bat")
    assert bat.name == "bat_room_timing.json"
    below = default_tip_room_timing_path("below_spazer")
    assert below.name == "below_spazer_room_timing.json"
    warehouse = default_tip_room_timing_path("warehouse")
    assert warehouse.name == "warehouse_room_timing.json"


def test_default_tip_artifact_paths_per_milestone() -> None:
    video, report = default_tip_artifact_paths("supers")
    assert video.name == "supers.mp4"
    assert report.name == "supers.json"


def test_run_to_rejects_room_timing_on_early_tip() -> None:
    with pytest.raises(ValueError, match="room timing"):
        run_to("morph", room_timing_path="/tmp/x.json")


def test_run_supers_accepts_room_timing_path() -> None:
    sig = inspect.signature(run_supers)
    assert "room_timing_path" in sig.parameters
    assert sig.parameters["room_timing_path"].default is None
    assert "tip" in inspect.signature(run_to).parameters


def test_checkpoint_output_is_explicit_and_early_tips_reject_it() -> None:
    from super_metroid.routes.catalog import get_continuous_tip

    sig = inspect.signature(run_tip)
    assert "state_output" in sig.parameters
    assert get_continuous_tip("varia").supports_checkpoint is True
    assert get_continuous_tip("business").supports_checkpoint is True
    assert get_continuous_tip("frog").supports_checkpoint is True
    assert get_continuous_tip("supers").supports_checkpoint is False
    with pytest.raises(ValueError, match="checkpoint output"):
        run_to("supers", state_output="/tmp/not-a-source.state")


def test_early_tip_specs_cover_morph_through_supers() -> None:
    """Early tips are TipSpec rows with real hops + finish plugins."""
    from super_metroid.routes.catalog import CONTINUOUS_SEGMENTS
    from super_metroid.routes.continuous import EARLY_TIP_BY_ID, EARLY_TIP_SPECS, TIP_BY_ID
    from super_metroid.routes.early_continuous import (
        play_bombs,
        play_morph,
        play_spore,
        play_supers,
    )
    from super_metroid.routes.kpdr.early_post_morph import (
        BOMBS_SPINE,
        SPORE_SPINE,
        SUPERS_SPINE,
    )
    from super_metroid.routes.kpdr.early_spine import MORPH_SPINE
    from super_metroid.routes.tips import TipSpec

    expected = ("morph", "bombs", "spore", "supers")
    assert tuple(s.tip_id for s in EARLY_TIP_SPECS) == expected
    assert set(EARLY_TIP_BY_ID) == set(expected)
    for tip_id in expected:
        assert isinstance(EARLY_TIP_BY_ID[tip_id], TipSpec)
        assert tip_id in TIP_BY_ID
        # Hop-composed: real SpineHop deltas; finish plugins shape reports.
        assert EARLY_TIP_BY_ID[tip_id].hops
        assert EARLY_TIP_BY_ID[tip_id].final_conditions_fn is not None
    # Parent chain + spines (morph is root).
    assert EARLY_TIP_BY_ID["morph"].parent_tip_id is None
    assert EARLY_TIP_BY_ID["bombs"].parent_tip_id == "morph"
    assert EARLY_TIP_BY_ID["spore"].parent_tip_id == "bombs"
    assert EARLY_TIP_BY_ID["supers"].parent_tip_id == "spore"
    # Hops are the early spines; assist + condition plugins shape finish_report.
    assert EARLY_TIP_BY_ID["morph"].hops is MORPH_SPINE
    assert EARLY_TIP_BY_ID["bombs"].hops is BOMBS_SPINE
    assert EARLY_TIP_BY_ID["spore"].hops is SPORE_SPINE
    assert EARLY_TIP_BY_ID["supers"].hops is SUPERS_SPINE
    assert EARLY_TIP_BY_ID["morph"].assist_mode == "ammo"
    assert EARLY_TIP_BY_ID["bombs"].assist_mode == "ammo"
    assert EARLY_TIP_BY_ID["spore"].assist_mode == "resources"
    assert EARLY_TIP_BY_ID["supers"].assist_mode == "resources"
    assert EARLY_TIP_BY_ID["morph"].emit_flat_video_path is True
    assert EARLY_TIP_BY_ID["bombs"].schema_version == 2
    assert EARLY_TIP_BY_ID["morph"].required_splits == ("morph_ball",)
    assert EARLY_TIP_BY_ID["supers"].success_outcome == "spore_supers_collected"
    # Segment registry binds public play_* wrappers (hop-composed).
    assert CONTINUOUS_SEGMENTS["morph"] is play_morph
    assert CONTINUOUS_SEGMENTS["bombs"] is play_bombs
    assert CONTINUOUS_SEGMENTS["spore"] is play_spore
    assert CONTINUOUS_SEGMENTS["supers"] is play_supers


def test_early_play_wrappers_delegate_to_spine_hops() -> None:
    """play_* wrappers target tip ids; spines carry multi-split after hooks."""
    from super_metroid.routes import early_continuous as early
    from super_metroid.routes.kpdr import early_post_morph as post
    from super_metroid.routes.kpdr import early_spine as morph_spine

    # Public early play_* wrappers bind a tip_id for play_tip (not hand-rolled hops).
    for tip_id, play_fn in (
        ("morph", early.play_morph),
        ("bombs", early.play_bombs),
        ("spore", early.play_spore),
        ("supers", early.play_supers),
    ):
        assert tip_id in play_fn.__code__.co_consts
        params = list(inspect.signature(play_fn).parameters)
        assert params[:2] == ["session", "splits"]
        assert "segments" in params
    # Pure-probe hop runners + spines remain; multi-split bookkeeping on hop.after.
    assert callable(post.play_bombs_hops)
    assert callable(post.play_spore_hops)
    assert callable(post.play_supers_hops)
    assert callable(morph_spine.play_morph_hops)
    assert len(post.BOMBS_SPINE) == 5
    assert len(post.SPORE_SPINE) == 2
    assert len(post.SUPERS_SPINE) == 1
    assert post.BOMBS_SPINE[0].after is not None  # two_missile multi-splits
    assert post.BOMBS_SPINE[4].after is not None  # bomb torizo multi-splits
    assert post.SPORE_SPINE[0].after is not None
    assert post.SPORE_SPINE[1].after is not None
    assert post.SUPERS_SPINE[0].after is not None
    # Hop room chains are contiguous across early spines.
    for spine in (morph_spine.MORPH_SPINE, post.BOMBS_SPINE, post.SPORE_SPINE):
        for prev, hop in zip(spine, spine[1:]):
            assert prev.to_room == hop.from_room, (
                f"{prev.hop_id} → {hop.hop_id}: room gap "
                f"{prev.to_room:#x} vs {hop.from_room:#x}"
            )


def test_unified_tip_specs_cover_full_chain() -> None:
    """One TipSpec table: early + Super+; red_tower parents to supers."""
    from super_metroid.routes.continuous import (
        SUPER_TIP_BY_ID,
        SUPER_TIP_SPECS,
        TIP_BY_ID,
        TIP_SPECS,
    )
    from super_metroid.routes.tips import TipSpec

    expected_super = (
        "red_tower",
        "bat",
        "below_spazer",
        "warehouse",
        "hijump",
        "kraid",
        "varia",
        "business",
        "frog",
        "bat_cave",
        "speed",
        "wave",
        "ice",
    )
    assert tuple(s.tip_id for s in SUPER_TIP_SPECS) == expected_super
    # Unified table includes early + Super+.
    for tip_id in ("morph", "supers", "red_tower", "bat_cave", "speed", "wave", "ice"):
        assert tip_id in TIP_BY_ID
        assert isinstance(TIP_BY_ID[tip_id], TipSpec)
    assert {s.tip_id for s in TIP_SPECS} >= set(expected_super) | {
        "morph",
        "bombs",
        "spore",
        "supers",
    }
    # Parent chain: red_tower → supers (early); frog + bat_cave are business siblings.
    assert SUPER_TIP_BY_ID["red_tower"].parent_tip_id == "supers"
    assert SUPER_TIP_BY_ID["bat"].parent_tip_id == "red_tower"
    assert SUPER_TIP_BY_ID["frog"].parent_tip_id == "business"
    assert SUPER_TIP_BY_ID["frog"].require_varia is True
    assert SUPER_TIP_BY_ID["bat_cave"].parent_tip_id == "business"
    assert SUPER_TIP_BY_ID["bat_cave"].final_room == 0xB07A
    assert SUPER_TIP_BY_ID["bat_cave"].hops  # hop-composed Super+ tip
    assert [h.split_id for h in SUPER_TIP_BY_ID["bat_cave"].hops] == [
        "business_to_cathedral_entrance",
        "cathedral_entrance_to_cathedral",
        "cathedral_to_rising_tide",
        "rising_tide_to_bubble",
        "bubble_to_bat_cave",
    ]
    assert SUPER_TIP_BY_ID["speed"].parent_tip_id == "bat_cave"
    assert SUPER_TIP_BY_ID["speed"].final_room == 0xAD1B
    assert [h.split_id for h in SUPER_TIP_BY_ID["speed"].hops] == [
        "bat_cave_to_speed_hall",
        "speed_hall_to_speed",
    ]
    assert SUPER_TIP_BY_ID["wave"].parent_tip_id == "speed"
    assert SUPER_TIP_BY_ID["wave"].final_room == 0xADDE
    assert [h.split_id for h in SUPER_TIP_BY_ID["wave"].hops] == [
        "speed_return_to_bubble",
        "bubble_to_single_chamber",
        "single_to_double_chamber",
        "double_chamber_to_wave",
    ]
    assert SUPER_TIP_BY_ID["ice"].parent_tip_id == "wave"
    assert SUPER_TIP_BY_ID["ice"].final_room == 0xA890
    assert SUPER_TIP_BY_ID["ice"].success_outcome == "ice_collected"
    # Wave→Business return (rr-vqv3) + Ice pure stack (rr-dbu.11); rr-kxge compose.
    assert [h.split_id for h in SUPER_TIP_BY_ID["ice"].hops] == [
        "wave_to_double_chamber",
        "double_to_single_chamber",
        "single_to_bubble",
        "bubble_to_farm",
        "farm_to_speedway",
        "speedway_to_frog_save",
        "frog_save_to_business",
        "business_to_ice_gate",
        "ice_gate_to_acid",
        "ice_acid_to_snake",
        "ice_snake_to_ice",
    ]
    # Catalog alias + default tip is Ice after rr-kxge dual continuous green.
    assert get_continuous_tip("ice_beam").tip_id == "ice"
    assert get_continuous_tip("k4_11").tip_id == "ice"
    assert DEFAULT_CONTINUOUS_TIP == "ice"


def test_post_supers_aliases_and_hop_tables() -> None:
    """Super+ tips go through play_tip; hop tables live on hops."""
    from super_metroid.routes import continuous as cont
    from super_metroid.routes.catalog import CONTINUOUS_SEGMENTS
    from super_metroid.routes.kpdr import hops as hop_mod
    from super_metroid.routes.tips import TipSpec

    assert callable(CONTINUOUS_SEGMENTS["warehouse"])
    assert callable(CONTINUOUS_SEGMENTS["red_tower"])
    assert CONTINUOUS_SEGMENTS["warehouse"].__name__ == "play_warehouse"
    # No per-tip play_/run_ aliases on the continuous module.
    assert not hasattr(cont, "play_red_tower")
    assert not hasattr(cont, "run_bat_cave")
    assert not hasattr(cont, "play_warehouse")
    # Hop tables are owned by hops.py (not re-exported from continuous).
    assert not hasattr(cont, "WAREHOUSE_HOPS")
    assert not hasattr(cont, "BAT_CAVE_ONLY_HOPS")
    assert not hasattr(hop_mod, "_WAREHOUSE_HOPS")
    assert not hasattr(hop_mod, "RouteHop")
    assert not hasattr(hop_mod, "PostSupersTipSpec")
    assert isinstance(hop_mod.WAREHOUSE_HOPS[0], hop_mod.SpineHop)
    assert hop_mod.SUPER_TIP_BY_ID["warehouse"].hops is hop_mod.WAREHOUSE_HOPS
    assert all(isinstance(s, TipSpec) for s in hop_mod.SUPER_TIP_SPECS)


def test_post_supers_report_kind_keeps_evidence_fields() -> None:
    """All kinds serialize the full schema; boss/super_collect present (null ok)."""
    report = ContinuousRunReport(
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


def test_report_to_dict_is_kind_agnostic() -> None:
    """morph/bombs/spore/supers all emit the same top-level key set."""
    base = dict(
        schema_version=1,
        success=False,
        outcome="failed:test",
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
        integrity=None,
        route_plan=None,
        policy_sources=None,
        state_loads=0,
        progression_writes=0,
        video=None,
        video_path=None,
        source_policy="test",
        rom_sha256="",
        start_state="power_on",
        generated_at="",
    )
    keys_by_kind: dict[str, set[str]] = {}
    for kind in ("morph", "bombs", "spore", "supers", "warehouse", "bat_cave"):
        payload = ContinuousRunReport(**base, kind=kind).to_dict()
        keys_by_kind[kind] = set(payload)
        assert payload["boss"] is None
        assert payload["super_collect"] is None
        assert payload["route_plan"] == {}
        assert payload["policy_sources"] == {}
        assert "video_path" in payload
        assert "error" in payload
        assert "integrity" in payload
    reference = keys_by_kind["supers"]
    for kind, keys in keys_by_kind.items():
        assert keys == reference, f"{kind} keys differ from supers"


def test_warehouse_hops_table_shape() -> None:
    from super_metroid.routes.kpdr.hops import WAREHOUSE_HOPS

    assert [h.split_id for h in WAREHOUSE_HOPS] == [
        "below_spazer_to_west",
        "west_to_glass",
        "glass_to_east",
        "east_to_warehouse",
    ]
    assert WAREHOUSE_HOPS[-1].to_room == 0xA6A1


def test_split_for_transition_uses_latest_repeated_doorway() -> None:
    transitions = [
        ObservedTransition(100, 0xA6A1, 0xA7DE, "warehouse_to_business"),
        ObservedTransition(200, 0xA6A1, 0xA7DE, "warehouse_to_business"),
    ]
    split = split_for_transition(
        transitions, "warehouse_to_business_return", 0xA6A1, 0xA7DE
    )
    assert split.frame == 200


def test_controller_module_exists() -> None:
    assert CONTROLLER_PATH.is_file()


def test_supers_report_includes_super_collect_field() -> None:
    # Smoke: dataclass accepts super_collect=None for failed early exits.
    report = ContinuousRunReport(
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
