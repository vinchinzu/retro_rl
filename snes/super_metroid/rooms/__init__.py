"""Full-room graph, problem catalog, and practice loop."""

from super_metroid.rooms.capabilities import normalize_ability
from super_metroid.rooms.room_graph import (
    load_problem_catalog,
    problem_by_id,
    shortest_room_path,
)
from super_metroid.rooms.segment_contract import EntryContract, resolve_entry_door_ptr
from super_metroid.rooms.work_queue import build_work_queue, difficulty_score

__all__ = [
    "EntryContract",
    "build_work_queue",
    "difficulty_score",
    "load_problem_catalog",
    "normalize_ability",
    "problem_by_id",
    "resolve_entry_door_ptr",
    "shortest_room_path",
]
