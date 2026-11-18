"""Map JSON lint: load, unique door labels, path points, approach extent."""

from __future__ import annotations

from alttp.room_map import KnownDoor, RoomMap, RoomMapPoint
from alttp.scripts.room_engine import lint_maps, lint_room, main


def _room(
    *,
    points: tuple[RoomMapPoint, ...] = (),
    doors: tuple[KnownDoor, ...] = (),
    walk_bbox: tuple[int, int, int, int] | None = None,
) -> RoomMap:
    return RoomMap(
        room_base_id=1,
        name="fixture",
        points=points,
        doors=doors,
        walk_bbox=walk_bbox,
    )


def test_shipped_maps_lint_clean() -> None:
    assert lint_maps() == []
    assert main(["lint"]) == 0


def test_named_point_outside_walk_bbox_is_inside_the_room() -> None:
    """room_61 west / room_82 west: approach is a point past walkBbox."""
    room = _room(
        points=(RoomMapPoint("west_door_approach", 520, 3320, role="approach"),),
        doors=(
            KnownDoor(
                label="west_to_0x60",
                direction="LEFT",
                approach_xy=(520, 3320),
                path=("west_door_approach",),
            ),
        ),
        walk_bbox=(576, 3176, 960, 3520),
    )
    assert lint_room("room_61.json", room) == []


def test_seam_allows_room_62_north_and_rejects_farther() -> None:
    room = _room(
        points=(RoomMapPoint("north_area", 1272, 3104),),
        doors=(
            KnownDoor(
                label="north_to_0x52",
                direction="UP",
                approach_xy=(1144, 3100),
                path=("north_area",),
            ),
        ),
        walk_bbox=(1023, 3104, 1328, 3360),
    )
    assert lint_room("room_62.json", room) == []
    far = _room(
        points=room.points,
        doors=(
            KnownDoor(
                label="north_to_0x52",
                direction="UP",
                approach_xy=(1144, 3091),
                path=("north_area",),
            ),
        ),
        walk_bbox=room.walk_bbox,
    )
    problems = lint_room("room_62.json", far)
    assert len(problems) == 1
    assert "outside room" in problems[0]


def test_duplicate_door_label_and_missing_path_point() -> None:
    room = _room(
        points=(RoomMapPoint("a", 0, 0),),
        doors=(
            KnownDoor(
                label="same",
                direction="UP",
                approach_xy=(0, 0),
                path=("missing",),
                recovery_path=("also_missing",),
            ),
            KnownDoor(label="same", direction="DOWN", approach_xy=(0, 0), path=("a",)),
        ),
    )
    problems = lint_room("room_xx.json", room)
    assert any("not unique" in line for line in problems)
    assert any("path point 'missing'" in line for line in problems)
    assert any("recoveryPath point 'also_missing'" in line for line in problems)


def test_bad_json_fails_load(tmp_path) -> None:
    (tmp_path / "room_bad.json").write_text("{", encoding="utf-8")
    problems = lint_maps(tmp_path)
    assert len(problems) == 1
    assert problems[0].startswith("room_bad.json: load failed:")


def test_no_extent_rejects_approach() -> None:
    room = _room(
        doors=(KnownDoor(label="out", direction="UP", approach_xy=(10, 10)),)
    )
    problems = lint_room("room_empty.json", room)
    assert len(problems) == 1
    assert "outside room none" in problems[0]
