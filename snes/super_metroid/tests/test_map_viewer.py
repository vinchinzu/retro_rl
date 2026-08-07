"""Area-basemap CoG path tests (pixel alignment + segment safety)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_metroid.map_viewer.coords import (
    MAP_SCREEN_PX,
    area_bounds,
    load_room_index,
    point_from_sample,
    to_area,
)
from super_metroid.map_viewer.paths import (
    export_path,
    load_continuous_report,
    load_human_task,
    load_path_source,
    load_series_jsonl,
    points_from_samples,
    segment_points,
)
from super_metroid.paths import FULL_ROOM_GRAPH_PATH, GAME_DIR, RECORDINGS_DIR

LANDING = 0x91F8
PARLOR = 0x92FD


@pytest.fixture(scope="module")
def rooms():
    if not FULL_ROOM_GRAPH_PATH.is_file():
        pytest.skip("full_room_graph.json not present")
    return load_room_index()


@pytest.fixture(scope="module")
def bounds(rooms):
    return area_bounds(rooms)


def test_crateria_bounds_match_legacy_png(rooms, bounds):
    b = bounds["Crateria"]
    # maps/legacy/crateria.png is 13056×4864 when present
    assert b.width_px == (b.max_map_x - b.min_map_x) * MAP_SCREEN_PX
    assert b.min_map_x == 6
    assert b.min_map_y == 0
    legacy = GAME_DIR / "maps" / "legacy" / "crateria.png"
    if legacy.is_file():
        from PIL import Image

        im = Image.open(legacy)
        assert im.size == (b.width_px, b.height_px)


def test_to_area_landing(rooms, bounds):
    room = rooms[LANDING]
    b = bounds["Crateria"]
    ax, ay = to_area(room, b, 1153, 1088)
    assert ax == (23 - 6) * 256 + 1153
    assert ay == 1088


def test_subpixel(rooms, bounds):
    room = rooms[LANDING]
    b = bounds["Crateria"]
    ax, ay = to_area(room, b, 100, 200, x_sub=0x8000, y_sub=0)
    assert ax == pytest.approx((23 - 6) * 256 + 100.5)
    assert ay == 200


def test_skip_offmap(rooms, bounds):
    assert (
        point_from_sample(
            rooms, bounds, room_id=LANDING, x=65000, y=100, frame=0
        )
        is None
    )


def test_segment_breaks_on_room_and_jump(rooms, bounds):
    samples = [
        {"room_id": PARLOR, "x": 100, "y": 200, "frame": 0},
        {"room_id": PARLOR, "x": 110, "y": 200, "frame": 1},
        {"room_id": PARLOR, "x": 500, "y": 200, "frame": 2},  # big jump
        {"room_id": LANDING, "x": 100, "y": 100, "frame": 3},
        {"room_id": LANDING, "x": 105, "y": 100, "frame": 4},
    ]
    pts = points_from_samples(rooms, bounds, samples)
    segs = segment_points(pts, max_step_px=48)
    # parlor short step, then jump breaks, then landing short
    assert len(segs) == 2
    assert all(s.room_id == segs[0].room_id for s in [segs[0]])
    assert segs[0].room_id == PARLOR
    assert segs[1].room_id == LANDING
    for seg in segs:
        for a, b in zip(seg.points, seg.points[1:]):
            assert ((a.ax - b.ax) ** 2 + (a.ay - b.ay) ** 2) ** 0.5 <= 48


def test_export_series(tmp_path, rooms, bounds):
    series = tmp_path / "series.jsonl"
    rows = [
        {"frame": 10, "room_id": LANDING, "x": 100, "y": 200, "x_sub": 0, "y_sub": 0},
        {"frame": 11, "room_id": LANDING, "x": 110, "y": 200, "x_sub": 0, "y_sub": 0},
    ]
    series.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    wp = load_series_jsonl(series, rooms, bounds, path_id="t1")
    assert wp.kind == "tas_series"
    assert len(wp.segments) == 1
    assert wp.primary_area == "Crateria"
    out = export_path(wp, tmp_path / "t1.json")
    data = json.loads(out.read_text())
    assert data["schema"] == "super_metroid_area_path_v2"
    assert data["segments"][0]["points"][0]["ax"]


def test_human_task_segments(rooms, bounds):
    path = GAME_DIR / "tasks" / "parlor_left_human.json"
    if not path.is_file():
        pytest.skip("no parlor human task")
    wp = load_human_task(path, rooms, bounds, stride=4, max_points=500)
    assert wp.kind == "human_trace"
    assert wp.segments
    assert wp.primary_area == "Crateria"
    # all segment points stay in parlor-ish pixel band of crateria map
    b = bounds["Crateria"]
    for seg in wp.segments:
        for p in seg.points:
            assert 0 <= p.ax <= b.width_px
            assert 0 <= p.ay <= b.height_px


def test_continuous_is_markers_only(rooms, bounds):
    path = RECORDINGS_DIR / "wave.json"
    if not path.is_file():
        pytest.skip("no wave report")
    wp = load_continuous_report(path, rooms, bounds)
    assert wp.kind == "continuous_sparse"
    assert wp.segments == []
    assert len(wp.markers) > 5
    auto = load_path_source(path, rooms, bounds)
    assert auto.kind == "continuous_sparse"
    assert auto.segments == []
