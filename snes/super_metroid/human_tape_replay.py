"""Shim: hop-level open-loop replay API lives in ``super_metroid.human_tape``.

Prefer ``from super_metroid.human_tape import …`` or submodules
(``human_tape.replay``, ``human_tape.midpoints``, …). This module re-exports
the former flat ``human_tape_replay`` surface for existing scripts/tests.
"""

from __future__ import annotations

from super_metroid.human_tape.anchors import (
    _ANCHOR_KIND_RANK,
    as_xy,
    load_anchors_index,
    match_anchor,
    parse_room_id,
    resolve_anchor_path,
)
from super_metroid.human_tape.hops import (
    HopSlice,
    load_task_json,
    resolve_hop_slice,
    resolve_hop_slice_typed,
)
from super_metroid.human_tape.midpoints import (
    lockstep_scan,
    materialize_lockstep_mid,
    propose_trace_midpoints,
)
from super_metroid.human_tape.replay import (
    check_hop_green,
    frame_action,
    iter_hop_steps,
    replay_hop,
    resolve_assist,
    run_hop_replay,
)

# Private aliases used by older internal imports
_parse_room_id = parse_room_id
_as_xy = as_xy

__all__ = [
    "HopSlice",
    "_ANCHOR_KIND_RANK",
    "_as_xy",
    "_parse_room_id",
    "as_xy",
    "check_hop_green",
    "frame_action",
    "iter_hop_steps",
    "load_anchors_index",
    "load_task_json",
    "lockstep_scan",
    "match_anchor",
    "materialize_lockstep_mid",
    "parse_room_id",
    "propose_trace_midpoints",
    "replay_hop",
    "resolve_anchor_path",
    "resolve_assist",
    "resolve_hop_slice",
    "resolve_hop_slice_typed",
    "run_hop_replay",
]
