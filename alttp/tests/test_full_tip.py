"""Offline contracts for the clean tip runner and dungeon-edge composition."""

from __future__ import annotations

from types import SimpleNamespace

from alttp.opening_route import castle_dungeon
from alttp.opening_route.castle_dungeon import MAIN_HALL_TO_NW_PREFIX
from alttp.opening_route.full_tip import run_to_verified_tip
from alttp.opening_route.segment import SegmentEvidence
from alttp.ram import (
    HYRULE_CASTLE_MAIN_HALL_ROOM,
    HYRULE_CASTLE_MAIN_WEST_ROOM,
    HYRULE_CASTLE_NW_ROOM,
    AlttpSnapshot,
)
from alttp.route_report import SegmentResult
from alttp.startup import StartupResult


def _snap(
    *,
    room: int = 0,
    indoors: bool = True,
    screen: int = 0,
    sword: int = 1,
) -> AlttpSnapshot:
    return AlttpSnapshot(
        game_mode=0x07 if indoors else 0x09,
        submodule=0,
        room_id=room,
        indoors=indoors,
        screen_id=screen,
        link_x=760,
        link_y=3320,
        link_direction=0,
        link_action=0,
        camera_x=0,
        camera_y=0,
        dark_world=False,
        sword_level=sword,
        lamp_level=1,
        num_keys=0,
        follower=0,
    )


class _FakeSegment:
    def __init__(
        self, segment_id: str, snapshot: AlttpSnapshot, acceptance: dict[str, bool]
    ):
        self.segment_id = segment_id
        self.snapshot = snapshot
        self.acceptance = acceptance
        self.sources: list[str] = []
        self.exit = SimpleNamespace(
            acceptance_keys=tuple(acceptance),
            require_all=True,
        )

    def play_checked(self, _env: object, *, source: str) -> SegmentEvidence:
        self.sources.append(source)
        return SegmentEvidence(
            segment_id=self.segment_id,
            ok=True,
            frames=10,
            snapshot=self.snapshot,
            source=source,
            phase="complete",
            acceptance=self.acceptance,
        )


def test_verified_tip_runner_composes_only_registered_continuous_segments() -> None:
    grounds = _snap(indoors=False, screen=0x1B, sword=0)
    hall = _snap(room=HYRULE_CASTLE_MAIN_HALL_ROOM)
    segments = {
        "castle_to_sword": _FakeSegment(
            "castle_to_sword", _snap(room=0x55), {"sword": True}
        ),
        "sword_to_secret_entrance_clear": _FakeSegment(
            "sword_to_secret_entrance_clear",
            _snap(indoors=False, screen=0x1B),
            {"outside": True},
        ),
        "pocket_to_main_hall": _FakeSegment(
            "pocket_to_main_hall", hall, {"main_hall": True}
        ),
        "castle_dungeon_prefix": _FakeSegment(
            "castle_dungeon_prefix",
            _snap(room=HYRULE_CASTLE_NW_ROOM),
            {"northwest_0x50": True},
        ),
    }

    result = run_to_verified_tip(
        object(),
        boot_fn=lambda _env, **_kwargs: StartupResult("castle", grounds, 5),
        get_segment_fn=segments.__getitem__,  # type: ignore[arg-type]
        settle_fn=lambda _env: SimpleNamespace(
            ok=True,
            frames=0,
            snapshot=_snap(room=HYRULE_CASTLE_NW_ROOM),
            reason="ready",
        ),  # type: ignore[arg-type]
    )

    assert result.ok is True
    assert result.phase == "verified_tip_reached"
    assert result.tip_node == "room_50"
    assert result.frames == 45
    assert [e.segment_id for e in result.segments] == list(segments)
    assert all(segment.sources == ["natural_boot"] for segment in segments.values())
    report = result.to_report()
    assert report["clean_chain"] is True
    assert report["verifiedTip"] == "room_50"


def test_verified_tip_runner_fails_fast_on_segment_exit_contract() -> None:
    grounds = _snap(indoors=False, screen=0x1B, sword=0)
    bad = _FakeSegment("castle_to_sword", _snap(room=0x55), {"fighter_sword": False})
    never = _FakeSegment("never", _snap(room=HYRULE_CASTLE_MAIN_HALL_ROOM), {"x": True})

    def lookup(segment_id: str) -> _FakeSegment:
        return bad if segment_id == "castle_to_sword" else never

    result = run_to_verified_tip(
        object(),
        boot_fn=lambda _env, **_kwargs: StartupResult("castle", grounds, 5),
        get_segment_fn=lookup,  # type: ignore[arg-type]
        settle_fn=lambda _env: SimpleNamespace(
            ok=True,
            frames=0,
            snapshot=_snap(room=HYRULE_CASTLE_NW_ROOM),
            reason="ready",
        ),  # type: ignore[arg-type]
        segment_ids=("castle_to_sword", "never"),
    )

    assert result.ok is False
    assert result.phase == "castle_to_sword"
    assert "fighter_sword" in result.blocker
    assert bad.sources == ["natural_boot"]
    assert never.sources == []


def test_castle_dungeon_prefix_composes_room_edges_in_order(monkeypatch) -> None:
    class _Env:
        snap = _snap(room=HYRULE_CASTLE_MAIN_HALL_ROOM)

    env = _Env()
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(castle_dungeon, "snapshot_env", lambda _env: env.snap)

    def fake_edge(
        _env: object,
        map_id: str,
        door_label: str,
        **_kwargs: object,
    ) -> SegmentResult:
        edge = next(
            edge
            for edge in MAIN_HALL_TO_NW_PREFIX
            if edge.map_id == map_id and edge.door_label == door_label
        )
        calls.append((map_id, door_label))
        env.snap = _snap(room=edge.target_room)
        return SegmentResult(
            ok=True,
            phase=f"via_{door_label}",
            frames=7,
            snapshot=env.snap,
        )

    monkeypatch.setattr(castle_dungeon, "run_room_edge", fake_edge)
    result = castle_dungeon.run_from_main_hall(env, source="test")

    assert result.ok is True
    assert result.phase == "castle_dungeon_prefix_complete"
    assert result.frames == 14
    assert result.snapshot.room_base_id == HYRULE_CASTLE_NW_ROOM
    assert result.acceptance["northwest_0x50"] is True
    assert calls == [("room_61", "west_to_0x60"), ("room_60", "north_to_0x50")]


def test_castle_dungeon_prefix_rejects_wrong_room_before_actions(monkeypatch) -> None:
    class _Env:
        snap = _snap(room=HYRULE_CASTLE_MAIN_WEST_ROOM)

    env = _Env()
    monkeypatch.setattr(castle_dungeon, "snapshot_env", lambda _env: env.snap)
    monkeypatch.setattr(
        castle_dungeon,
        "run_room_edge",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = castle_dungeon.run_from_main_hall(env, source="test")

    assert result.ok is False
    assert result.phase == "entry_not_room_61"
    assert "expected room 0x61" in result.blocker
